from __builtin__ import staticmethod

__author__ = 'sergey, Lior'
import json
import logging
import cv2
import os
import urllib
import warnings
import collections
from .constants import db

# image is a structure for upload new tree function:
Image = collections.namedtuple("image", ["location", "url"])
# Bucket is a structure for answers classification ("key" - group id, "content" -
# all answers that relate to this group).
Bucket = collections.namedtuple("Bucket", ["key", "content"])
# TODO: raise an error if input is not suitable (do for each function)
# example : if isinstance(x, number_types) and isinstance(y, number_types): raise error

class CatNode(object):
    """
    Each Object has 7 Attributes:
    -name: This is the name of the Category
    -description: This is a text description of the Category
    -level: This is a Number telling us the level of the Category (Starting from root-Level as 1 to n sub levels)
    -image: This is the URL to an Image of the Category
    -id: This is a unique Number to identify a single Category
    -parent: This is the unique Number of the parent Object id (if the node is the child node, if it
    is a node on the root level the number is 0)
    -children: This is an Array of the child-nodes (sub nodes) modeling the parent-child relationship between nodes.
    -attributes: This is an array of attribute subtrees.
    04.07.15:
    To solve a problem of FindTruePath function (function obtains number of lists which describes "ways"
    in the obtainable tree and returns "the most right path to go" because it was chosen by most travelers).
    we must add the additional attribute which will say us if CatNode is marked as node of primary tree
    or an "attribute tree". This attribute is:
    - is_primary (equals to True if CatNode belongs to the primary tree, otherwise equals False
    (belongs to an "attribute tree").
    IMPORTANT!!!:
    Adding of new attribute will cause some changes in  __init__, copy and to_struct functions!!!
    23.07.15:
    For solve the problem of separation of nodes to different buckets in "GET_RIGHT_ANSWER" task we add
    a new attribute: "head".
    Each branch from root and to all leafs in CatNode tree has got one node that its level (depth) is
    sufficient to get an information about some product (clothing), more than that all answers
    (the paths of CatNode tree) will be classified to different groups by theirs node with head == true
    attribute...
    """
    id = 0
    cat_tree = None

    def __init__(self, t_type=None, level=None, **kwargs):
        """
        __init__ is a constructor of CatNode class.
        :param kwargs: dictionary of CatNode attributes: name, description, level,
         image, id, parent, children, attributes.
        :return: CatNode node
        """
        # If the obtainable dictionary does not contains "attributes" key => create in
        # CatNode empty "attributes" attribute otherwise create "attributes" attribute
        # equals to value of dictionary under "attributes" key (as similar as with "children" attribute):
        self.name = kwargs.get("name", "????????????????")
        # If t_type is not None that means that a current CatNode is node of primary tree,
        # otherwise a current CateNode is node of "attribute tree".
        if t_type is None:
            self.is_primary = True
        else:
            self.is_primary = False
        # The obtainable dictionary may contain "description" field or "desc" field:
        if kwargs.get("description") is not None:
            self.description = kwargs.get("description", "")
        else:
            self.description = kwargs.get("desc", "")
        # if the obtainable dictionary contains "head" field":
        if kwargs.get("tg_cat") is not None:
            self.head = True
        else:
            self.head = False
        self.parent = None
        self.image = kwargs.get("image", "!?!?!?!?!?!?!?!")
        # If user want set level by himself or when __init__ function
        # builds tree from dictionary (recursion) level its not None.
        if level:
            self.level = level
        else:
            self.level = self.parent.level + 1 if self.parent else 1
        self.id = kwargs.get("id", self.__class__.get_id())
        self.children = []
        self.attributes = []
        children = kwargs.get("children", [])
        attributes = kwargs.get("attributes", [])
        #  If item of key "children" is not empty:
        if children:
            for child in children:
                # If parent is not node of primary tree => his children also aren't.
                if self.is_primary is False:
                    is_attribute_t = True
                else:
                    is_attribute_t = None
                self.add_child(CatNode(t_type=is_attribute_t, level=self.level + 1, **child))
        # If item of key "attributes" is not empty
        if attributes:
            # (t_type =1) =>  To each root of "attribute tree" mark
            # that this root is not primary (self.is_primary=False)
            for attribute in attributes:
                self.add_attributes(CatNode(t_type=True, **attribute))

    @classmethod
    def get_id(cls):
        cls.id += 1
        return cls.id

    @classmethod
    def reset_id(cls):
        cls.id = 0

    def print_c_node(self):
        """
        The function prints all the attributes of the current tree.
        :return: None
        """
        print "--------------"
        print "id: " + str(self.id)
        print "name: " + self.name
        print "level: " + str(self.level)
        print "parent: " + str(self.parent)
        print "image url: " + self.image
        print "children: " + str(self.children)
        print "attributes: " + str(self.attributes)
        print "is head: " + str(self.head)
        print "is primary: " + str(self.is_primary)

    @staticmethod
    def dict_is_suitable(dictionary):
        #  Sergey:
        #  I have not used this function because the obtainable tree is not ideal
        # ( not all of nodes in the obtainable json structure contains all fields),
        #  And it is very important not to lose any nodes.
        if type(dictionary) is not dict:
            return False
        if dictionary.get("name") is not None and dictionary.get("description") is not None \
                and dictionary.get("children") is not None:
            return True
        else:
            return False

    def add_child(self, child):
        """
        add_child function inserts the obtainable node into
        child list of current CatNode
        :param child: child must be a CatNode
        :return:
        """
        if type(child) is not CatNode:
            raise TypeError("child must be CatNode")
        # child = child or CatNode(**kwargs)
        child.parent = self
        child.level = self.level + 1
        self.children.append(child)

    def add_attributes(self, attribute):
        """
        add_attribute function inserts the obtainable attribute
        into attribute list of current CatNode.
        :param attribute: attribute must be CatNode
        :return:
        """
        if type(attribute) is not CatNode:
            raise TypeError("Attribute must be CatNode")
        attribute.parent = self
        attribute.level = self.level + 1
        self.attributes.append(attribute)

    def __repr__(self):
        # return "<CatNode {0}, {1}>".format(self.id, self.name)
        return str(self.id)
        # return self.children

    def count_amount_of_primary_nodes(self):
        """
        TEMPORARY FUNCTION!!!
        The main purpose of the function is to check work of __init__ function after updating.
        The function counts number of nodes which belong to primary tree.
        is_primary attribute gives us an opportunity to know a status of each CatNode
        after an implementation of the complicate function.
        :return:
        """
        children = self.children
        is_prim = 0
        if self.is_primary is True:
            is_prim = 1
        return is_prim + sum((child.count_amount_of_primary_nodes() for child in self.children))

    def count_amount_of_attr_nodes(self):
        """
        TEMPORARY FUNCTION!!!
        The main purpose of the function is to check work of __init__ function after updating.
        The function counts number of nodes which do not belong to primary tree.
        is_primary attribute gives us an opportunity to know a status of each CatNode
        after an implementation of the complicate function.
        :return:
        """
        children = self.children
        is_attr = 1
        if self.is_primary is True:
            is_attr = 0
        return is_attr + sum((child.count_amount_of_attr_nodes() for child in self.children))

    def size(self):
        """
        The function counts number of nodes of the tree (without subtrees
        located in attributes).
        :return:
        """
        return 1 + sum((child.size() for child in self.children))

    def count_attributes(self):
        """
        The function returns number of "attribute subtrees" in current tree.
        :return:
        """
        return len(self.attributes) + sum((child.count_attributes() for child in self.children))

    # ----------------Block_of_build_tree_from_json_functions--------------------

    def connect_to_leafs(self, children):
        """
        The function obtains list of subtrees and connects each
        of them to each of current tree leafs.
        (example: after function implementation leaf.children[] contains each
        of subtrees from the obtainable children list)
        :param children: list of CatNode trees.
        :return:
        """
        #  while current node is not leaf:
        for child in self.children:
            child.connect_to_leafs(children)
        # Current node is leaf therefore we add to it the obtainable children
        # Add children only to node which has not got any child
        if not self.children:
            for child in children:
                #  Very important do not add child to all of leafs!!!
                #  At this vay you will only connect all tree leafs with single sub-tree!!!
                #  that's why we clone child sub-tree.
                child_copy = CatNode()
                self.add_child(child_copy.copy(child, self.level + 1))
                self.children[-1].correct_levels()

    def connect_sub_trees(self, attributes):
        """
        The function connect the obtainable subtrees to current node (multiplies them).
        BE CAREFUL: connect_sub_trees function does not connect roots of the obtainable
        subtrees (connects a subtree but without its root)!!! that is because the main algorithm "expand an attribute tree
        to a complicated tree" ignore roots of a attribute trees.
        WARNING: input must be formed from different trees if the function will obtain some tree more
        than one time it will cause to an infinity recursion.
        :param attributes: list of subtrees
        :return:
        """
        # If list of sub trees is not empty - do:
        if attributes:
            children = attributes.pop(0).children
            self.connect_to_leafs(children)
            self.connect_sub_trees(attributes)

    def complicate(self):
        """
        This function obtains tree.
        For each node in the tree:
            if node contains attributes list => add
            all of the subtrees in an attribute list to tree (subtree without root).
        after complicate function implementation all nodes of obtainable
        tree do not contain any attributes.
        all of the tree's attributes are a part of the tree now.
        :return:
        """
        attr = self.attributes
        if attr:
            # Clear attributes:
            self.attributes = []
            self_copy = CatNode()
            # Copy subtree of current node (with its root),
            # and paste it to the last place in attribute_list:
            attr.append(self_copy.copy(self))
            self.children = []  # Disconnect list of children from the current node.
            self.connect_sub_trees(attr)
        children = self.children
        if children:
            for child in children:
                child.complicate()

    def copy(self, node, level=None):
        """
        The function copies the obtainable "node" tree into current tree.
        :param node: tree which function will copy.
        :param level: number of root level.
        :return: copy of the obtainable tree.
        """
        #  Fill current tree node with attributes from obtainable node.
        self.is_primary = node.is_primary
        self.name = node.name
        self.description = node.description
        self.image = node.image
        self.head = node.head  # is it necessary to copy also a "head" status?
        self.attributes = []
        self.children = []
        if level is None:
            self.level = node.level
        else:
            self.level = level
        if node.children:
            for child in node.children:
                new_child = CatNode()
                self.add_child(new_child.copy(child, self.level + 1))
        if node.attributes:
            for attribute in node.attributes:
                new_attribute = CatNode()
                self.add_attributes(new_attribute.copy(attribute, self.level + 1))
        return self

    @staticmethod
    def from_str(json_string):
        """
        The function obtains dictionary with "specific structure" (like CatNode structure),
        and returns tree builded from the obtainable dictionary (List of sub category trees).
        :param json_string:
        :return list of CatNode trees:
        """
        main_structure = json.loads(json_string)
        # Do not forget: to change from "main_structure[0]" to "main_structure",
        #  because tis function obtains only class tat_trees!
        list_of_categories = []
        if main_structure.get("categories"):
            cat_dict = main_structure["categories"]
            # Build tree for all category:
            for category in cat_dict:
                list_of_categories.append(CatNode(**category))
        else:
            cat_dict = main_structure
            list_of_categories.append(CatNode(**cat_dict))
        return list_of_categories

    def to_struct(self, tree_dict):
        """
        The function converts "tree_dict" structure (type CatNode)
        to dictionary, and safes it into tree_dict.
        Warning: if tree_dict is not empty "to_struct" function
        will not delete its content.
        :param tree_dict:
        :return:
        """
        tree_dict["name"] = str(self.name)
        tree_dict["description"] = str(self.description)
        tree_dict["level"] = self.level
        tree_dict["id"] = self.id  # + 10000
        tree_dict["head"] = self.head
        if self.parent:
            tree_dict["parent"] = self.parent.id  # + 10000
        else:
            tree_dict["parent"] = 0
        tree_dict["image"] = str(self.image)
        if self.children:
            tree_dict["children"] = []
            children = self.children
            for child in children:
                new_dict = {}
                tree_dict["children"].append(child.to_struct(new_dict))
        if self.attributes:  # 02.08.15
            tree_dict["attributes"] = []
            attributes = self.attributes
            for attribute in attributes:
                new_dict2 = {}
                tree_dict["attributes"].append(attribute.to_struct(new_dict2))
        return tree_dict

    def to_js(self):
        """
        This function converts the obtainable tree to json tree.
        :return:
        """
        dict_tree = dict()
        py_struct = self.to_struct(dict_tree)
        return str(json.dumps(py_struct))

    @staticmethod
    def full_tree_to_js(root):
        """
        This function converts the obtainable tree to json tree.
        :param root:
        :return: list of subtrees.
        """
        root_dict = {}
        root_dict["categories"] = []
        for sub_root in root:
            dict_tree = dict()
            py_struct = sub_root.to_struct(dict_tree)
            root_dict["categories"].append(py_struct)
        # only for test:
        # pprint.pprint(root_dict)
        # return str(json.dumps(root_dict))
        return json.dumps(root_dict)

    def correct_levels(self):
        """
        the function rewrites levels in all nodes by logical order.
        :return:
        """
        children = self.children
        for child in children:
            child.level = self.level +1
            child.correct_levels()

        #  -----------------------------------Block_of _change_urls_in_tree-------------------------------------

    def apply_to_all_nodes(self, func, *arg):
        """
        This function gives an opportunity to correct the attributes of CatNodes of the tree.
        :param dict_attr:
        :return:
        """
        func(self, *arg)
        children = self.children
        if children:
            for child in children:
                child.apply_to_all_nodes(func, *arg)

    @staticmethod
    def is_only_name(str_url):
        """
        the function checks if the obtainable string is only an image name or full url.
        :param str_url:
        :return: boolean value
        """
        if "/" in str_url:
            return False
        return True

    @staticmethod
    def download_image(dir_url, destination_dir):
        """
        The function saves image from dir-url, in case if the url does not exists
        function will return false otherwise function will return True
        :param dir_url: urs from which the function will take the image
        :param destination_dir: directory where to save the image
        :return: boolean value
        """
        name = dir_url.split("/")[-1]
        try:
            status = urllib.urlretrieve(dir_url, name)
            img = cv2.imread(name, 0)
            # Resizing of an image:
            img = cv2.resize(img, (60, 60))
            # cv2.imwrite(destination_dir + "\\" + name, img)
            upload_image(img, name, destination_dir)
            try:
                os.remove(name)
            except:
                logging.warning("S: Image hadn't been deleted!!!")
        except:
            return False
        return True

    @staticmethod
    def get_name(str_url):
        """
        The function obtains url of the image, and returns only its number
        :param str_url:
        :return: name string
        """
        return str(str_url).split('/')[-1]

    @staticmethod
    def update_url(node, url_str, empty_image_values):
        """
        The function obtains CatNode, url string, list of strings,
        and corrects the self.image attribute.
        :param url_str: new url (without image name)
        :param empty_image_values: (if image is empty it has got a value by default)  list of strings.
        :return: None
        """
        current_url = node.image
        # Current url does not exist:
        if current_url in empty_image_values:
            logging.warning("error 00001: Image does not exist!!!")
        else:

            name = CatNode.get_name(current_url)
            if CatNode.is_only_name(current_url):
                node.image = url_str + name
            else:
                if CatNode.download_image(current_url, "C:\Users\sergey\Desktop\\tree_images\images"):
                    node.image = url_str + name
                else:
                    # Mark that current url isn't valid:
                    node.image = "???????????????????????"

    # ---Block_of_upload/edit_new_tree------------------------------------------------#
    @staticmethod
    def push_url(node, url, stack):
        if url != node.image[:len(url)]:
            stack.append(Image(location=node, url=node.image))

    @staticmethod
    def upload_new_tree(json_url, right_url, download_function, destination):
        """ The function obtains url to json tree, downloads it, checks it correctness,
        if the obtainable tree is correct update it.
        :param json_url: link to the json tree
        :param right_url: linc to the place where all tree's images must be located.
        :param download_function: The problem is that the download functions are not ideal, not
                universal to all cases. Therefore i decided to receive a download function as parameter.
        :param destination: url, the place where new images will be saved.
        :return: new, updated tree.
        """
        # Check if json tree exists:
        try:
            urllib.urlretrieve(json_url, "input_json.txt")
            with open("input_json.txt", "r") as f:
                json_tree = f.read()
                return CatNode.__upload_new_tree__(json_tree, right_url, download_function, destination)
                f.close()
        except:
            raise Exception("S: Json file was not found!!!")

    @staticmethod
    def __upload_new_tree__(json_tree, right_url, download_function, destination):
        """
        This function is a second part of upload_new_tree function. All description about its work
        is placed in upload_new_tree function's description.
        :param json_tree: tree_structured json string.
        :param right_url:
        :param download_function:
        :param destination:
        :return:
        """
        # Check if a json tree is valid (valid javaScript structure):
        try:
            py_tree = json.loads(json_tree)
        except:
            raise Exception("S: json is not valid!!!")
        # Check if each branch from root to leafs in the json tree has got a head node:
        cat_tree = CatNode.from_str(json_tree)
        for subtree in cat_tree:
            CatNode.cat_tree = subtree
            if not subtree.check_tree():
                raise Exception("S: Exists branch without 'head' node!!!")
        new_urls_list = []
        for subtree in cat_tree:
            subtree.apply_to_all_nodes(CatNode.push_url, right_url, new_urls_list)
            if len(new_urls_list) > 0:
                for img_tup in new_urls_list:
                    # if node.image is only name => this image is already placed in bucket,
                    # that means we will not download it.
                    if not CatNode.is_only_name(img_tup.url):
                        if download_function(img_tup.url, destination):
                            img_tup.location.image = right_url + CatNode.get_name(img_tup.url)
                        else:
                            logging.warning("S: The image " + img_tup.url + " was not downloaded")
        return cat_tree

    #  ------------------ block_of_find_right_answer_functions ---------------------------
    @staticmethod
    def head(id):
        """
        The function obtains a node id, and searches for name of "head node"
        :param id: CatNode id (string)
        :return: id of "head node" (string)
        """
        current = CatNode.cat_tree.find_by_id(id)
        # find a head node:
        while current and current.head is False:
            current = current.parent
        # If id wasn't suitable:
        if not current:
            return None
        return current.id

    @staticmethod
    def check_ans(ans_mat):
        """
        The function obtains list of answer-lists ("answers-matrix") and checks each answer.
        If some answer is not completed or duplicated (only in list) => function will clear it from ans_mat.
        :param ans_mat: matrix of strings (CatNode id-s).
        :return: None
        """
        res_mat = []
        # remove nodes if their tree branch doesn't contain "head node"
        for ans_list in ans_mat:
            for id in reversed(ans_list):
                if CatNode.head(id) is None:
                    ans_list.remove(id)
        # to leave only one id per one head:
        for ans_list in ans_mat:
            right_nodes_dict = {}
            for id in ans_list:
                right_nodes_dict[CatNode.head(id)] = id
            res_mat.append(right_nodes_dict.values())
            #  res_mat.append({head(id), id for id in ans_list}.values())
        return res_mat

    @staticmethod
    def build_bucket(id):
        """
        The function obtains answer (CatNode id) and builds bucket:
        bucket(key = "head" of the "branch" to which relates CatNode, CatNode)
        We can create bucket from id only if in a tree branch to which id belong presents only one CatNode that
        has got an attribute head == True.
        :param id: CatNode id.
        :return: bucket.
        """
        key = CatNode.head(id)
        if key is not None:
            buck = Bucket(key=key, content=[id])
            return buck
        logging.warning("error s: a bucket was not created!!!")
        return None

    @staticmethod
    def push_to_bucket(buck, ans_list):
        """
        The function obtains bucket, ans_list of CatNode id-s and pops from an answers
        list only a suitable id-s to the bucket.
        :param buck: bucket.
        :param ans_list: ans_list.
        :return: None.
        """
        for id in ans_list:
            if CatNode.head(id) == buck.key:
                buck.content.append(id)

    @staticmethod
    def determine_final_categories(ans_mat):
        """
        The function obtains "answer matrix" (list of answer-lists) which contains only answers
        about clothing of a single person from some image.
        Each "answer-list" describes answers that were done by only one person, about number of clothing groups
        (example: dresses, boots, bottoms etc...). Each item of "answer-list" is "id" (string) of CatNode
        node in category tree.
        This function calculates a single right answer for each clothing group (if it possible),
        and returns list of right answers (id-s) where each answer relates to some specific clothing group.
        :param ans_mat: matrix of strings (CatNode id-s).
        :return:list of strings (CatNode id-s).
        """
        ans_mat = CatNode.check_ans(ans_mat)
        print "ans_matrix:" + str(ans_mat)
        # find the longest answer-list:
        max_list = min(ans_mat, key=lambda ans_list: len(ans_list))
        bucket_list = [CatNode.build_bucket(node) for node in max_list]
        # Remove the processed list:
        ans_mat.remove(max_list)
        for ans_list in ans_mat:
            for bucket in bucket_list:
                CatNode.push_to_bucket(bucket, ans_list)
            # if after separation of ans_list elements ans_list is not empty
            # that means that we have not created all buckets! therefore create another buckets:
            if ans_list:
                for node in ans_list:
                    bucket_list.append(CatNode.build_bucket(node))
        # for each suitable bucket find answer:
        right_ans_list = []
        for bucket in bucket_list:
            if len(bucket.content) > 2:
                print "bucket" + str(bucket)
                right_ans_list.append(str(CatNode.cat_tree.find_ans(bucket.content)))
        print "right answers list: " + str(right_ans_list)
        return right_ans_list

    def find_ans(self, nodes_list):
        """
        The function obtains list of id-s, each id represents a specific answer.
        The obtainable list of answers contains only answers related to a single product
        (to one clothing item). find_ans function returns a single id from the obtainable list that
        is the most right answer relatively to others in the list.
        :param nodes_list: list of strings.
        :return:single string.
        """
        # tuples_list is a list of tuples where each one of them:
        # tuple(one of the current node children, list of nodes from nodes_list that relate to subtree
        # of current node (without current node) )
        tuples_list = [(child, child.find_by_id(nodes_list, True)) for child in self.children]
        # children node of current node which has got the biggest number of nodes from nodes_list
        # in its subtree will be placed in the end of Tuple_list:
        tuples_list.sort(key=lambda tup: len(tup[1]))
        biggest_tup = tuples_list.pop()
        # if tuples_list has got number of "biggest_tuples" => return current CatNode
        # because we cannot know which from them is right:
        prev_biggest_tuple = tuples_list.pop()
        max_len = len(biggest_tup[1])
        if max_len == len(prev_biggest_tuple[1]):
            return self
        # if under current node placed single node from node_list:
        if max_len == 1:
            return self
        # the case of special tuple:
        if (len(collections.Counter(biggest_tup[1])) == 1 and self.find_by_id(biggest_tup[1][0]).parent == self):
            return biggest_tup[0]
        elif max_len == 2:
            return CatNode.common_root(self.find_by_id(biggest_tup[1]))
        else:
            return biggest_tup[0].find_ans(biggest_tup[1])

    def root(self):
        """
        The function searches root of the current node, and returns it.
        :return: CatNode.
        """
        current = self
        if current:
            while current.parent:
                current = current.parent
        return current

    @staticmethod
    def __is_common__(node, nodes_list):
        """
        The function checks if a node's subtree contains all nodes from nodes_list.
        :param node:
        :param nodes_list:
        :return: Boolean (True if a node's subtree contains all nodes from nodes_list, otherwise False)
        """
        for each_node in nodes_list:
            # if one of list nodes doesn't relate to node subtree:
            if not node.find_by_id(each_node.id):
                return False
        return True

    @staticmethod
    def common_root(nodes_list):
        """
        The function obtains list of nodes from a common tree and searches
        the nearest common node (a common node may be one of the obtainable nodes).
        function will return the common node.
        WARNING: the function will not work if in the obtainable list of nodes at least one node
        relates to another tree. (ALL NODES MAST RELATE TO A SINGLE TREE!!!)
        :param nodes_list: list of nodes.
        :return: CatNode.
        """
        # If all the obtainable nodes relate to the same tree do, otherwise return None with warning.
        first_node = nodes_list[0]
        for node in nodes_list[1:]:
            if first_node.root() != node.root():
                warnings.warn("s: Incorrect input.")
                return None
        # find the highest node (node with min level)
        current = min(nodes_list, key=lambda v: v.level)
        while not CatNode.__is_common__(current, nodes_list):
            if current.parent:
                current = current.parent
            else:
                warnings.warn("s: Incorrect input.")
                return None
        return current

    def find_by_id(self, f_str, flag=None):
        """
        if type of f_str variable is list:
        The function scans all tree till it will find each CatNode
        (or will decide that this node does not present in the list) with id
        that equals to the string from f_str list.
        In case if function hadn't found any CatNode-s it will return an empty list.
        :param f_str: list of id-s by which the function will find a suitable CatNode-s
        :return: list of CatNode-s
        if type of f_str variable is string:
        The function scans all tree till it will find CatNode with id that equals to the obtainable string (f_str).
        In case if such a node was not found function will return None value.
        :param f_str: id by which the function will find a suitable CatNode
        :return:CatNode
        """
        def __fbi__(current, f_str):
            """
            The function searches in "current" tree node whose id attribute is equal to f_str, if such
            node is presented in the "current" tree function will save it into the obtainable variable: res_node
            :param current: CatNode
            :param f_str: the attribute that function searches in the "current" tree.
            :param res_node: the variable into which function will save the result.
            """
            # if we found the suitable node => stop search in the tree:
            if str(current.id) == str(f_str):
                return current
            else:
                children = current.children
                # if the current CatNode have got any children:
                if children:
                    for child in children:
                        if __fbi__(child, f_str):
                            return __fbi__(child, f_str)
            # in case node with f_str id does not exist:
            return None

        def __fbi_list__(current, f_str, result_list):
            """
            The function searches in "current" tree node whose id attribute is equal to f_str, if such
            node is presented in the "current" tree function will save it into the obtainable variable: res_node
            :param current: CatNode
            :param f_str: the attribute that function searches in the "current" tree.
            :param res_node: the variable into which function will save the result.
            """
            # if we found the suitable node => save in result_list:
            if str(current.id) in f_str:
                for id in f_str:
                    if id == str(current.id):
                        result_list.append(current)
                        # f_str.remove(str(current.id))
            children = current.children
            if children:
                for child in children:
                    __fbi_list__(child, f_str, result_list)

        # if type of the obtainable f_str variable is str => find a single node
        if type(f_str) is str or type(f_str) is int:
            node = __fbi__(self, f_str)
            if flag:
                return str(node.id)
            return node
        # if type of the obtainable f_str variable is list => find each node related
        # to id in te list (return list of nodes).
        temp_list = []
        if type(f_str) is list:
            result_list = []
            for id in f_str:
                temp_list.append(str(id))
            __fbi_list__(self, temp_list, result_list)
            # if flag is not None instead of return list nodes function will return list of their id-s:
            if flag:
                result_list = [str(node.id) for node in result_list]
            return result_list

    def build_path(self):
        """
        The function calculates the hole path from the tree root to the current node
        (include root and current node)
        :return: list of nodes which "describes" (builds) the path.
        """
        path_l = []
        current = self
        while current:
            path_l.append(current)
            current = current.parent
        path_l.reverse()
        return path_l

    @classmethod
    def get_tree(cls):
        """
        The function downloads tree (python structures) from a database,
        converts it to CatNode tree structure and sets it in cat_tree class variable
        and returns it.
        :return: CatNode tree.
        """
        # if cat_tree is not created:
        if CatNode.cat_tree is None:
            try:
                tree_dict = db.globals.find_one({"_id": "tg_globals"})["category_tree_dict"]
            except pymongo.errors.ServerSelectionTimeoutError:
                raise Exception("s: connection with server is failed ")
                return None
            return CatNode.__to_full_tree__(tree_dict)
        return None

    @staticmethod
    def __to_full_tree__(tree_dict):
        """
        the function converts the obtainable dictionary
        to CatNode tree.
        :param tree_dict: tree dictionary is dictionary that presences
               CatNode tree but in which root is not structured as CatNode
               (without any of CatNode attributes).
        :return:
        """
        # build root:
        c_tree = CatNode()
        c_tree.name = "categories"  # or will it be better to call "root"?
        c_tree.level = 0
        # connect root to subtrees:
        for sub_dict_tree in tree_dict['categories']:
            c_tree.children.append(CatNode(**sub_dict_tree))
        for child in c_tree.children:
            child.parent = c_tree
        CatNode.cat_tree = c_tree
        return CatNode.cat_tree

    def check_tree(self):
        """
        the function checks if per branch from root to each leaf exists CatNode
        That its head attribute equals to True.
        If at least one branch has not got "head" CatNode => function will return False
        otherwise True.
        :return: boolean value.
        """
        def __scan_for_head__(node, leafs):
            if not node.children:
                leafs.append(node)
            else:
                for child in node.children:
                    __scan_for_head__(child, leafs)
        leafs = []
        __scan_for_head__(self, leafs)
        for node in leafs:
            if CatNode.head(node.id) is None:
                print "here"
                print node.name
                return False
        return True

if __name__ == "__main__":
    tree = CatNode.get_tree()
